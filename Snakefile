# chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr20', 'chr21', 'chr22', 'chrX']
chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19']#, 'chrX']
# chroms = ['chr18']
# regions = [f'chr{i}_{j}' for i in range(1, 23) for j in ['p', 'q']] + ['chrX_p', 'chrX_q']
 
resolutions = [25000, 50000] #[100000, 50000, 25000, 10000]

rule all:
    input:
        expand("results_4Mar2024_kNN/notebook.000.joint-decomp.cis.{chrom}.{resolution}.nbconvert.ipynb", chrom=chroms, resolution=resolutions)

rule test_gpu:
    output:
        file="results_4Mar2024_kNN/notebook.000.joint-decomp.cis.{chrom}.{resolution}.nbconvert.ipynb",
    run:

        shell(r"""
            OUTPUT="./results_4Mar2024_kNN/notebook.000.joint-decomp.cis.{wildcards.chrom}.{wildcards.resolution}"
            cp /home/agalicina/IMMUNE/joint-decomposition/000.joint-decomp.cis.CHROM.RESOLUTION.kNN.ipynb $OUTPUT.ipynb
            sed -i 's/CHROM\=\\\"chr19\\\"/CHROM\=\\\"{wildcards.chrom}\\\"/g' $OUTPUT.ipynb
            sed -i 's/BINSIZE\=25000/BINSIZE\={wildcards.resolution}/g' $OUTPUT.ipynb
            jupyter nbconvert --to notebook --execute $OUTPUT.ipynb
            jupyter nbconvert --to html $OUTPUT.nbconvert.ipynb
            rm $OUTPUT.ipynb
        """)
        
