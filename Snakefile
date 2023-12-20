chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chrX']
# chroms = ['chr18']
resolutions = [100000, 50000, 25000, 10000]

rule all:
    input:
        expand("results/notebook.joint-decomp_cis.{chrom}.{resolution}.nbconvert.ipynb", chrom=chroms, resolution=resolutions)

rule test_gpu:
    output:
        file="results/notebook.joint-decomp_cis.{chrom}.{resolution}.nbconvert.ipynb",
    run:

        shell(r"""
            OUTPUT="./results/notebook.joint-decomp_cis.{wildcards.chrom}.{wildcards.resolution}"
            cp /home/agalicina/IMMUNE/joint-decomposition/joint-decomp_cis.CHROM.RESOLUTION.ipynb $OUTPUT.ipynb
            sed -i 's/CHROM\=\\\"chr19\\\"/CHROM\=\\\"{wildcards.chrom}\\\"/g' $OUTPUT.ipynb
            sed -i 's/BINSIZE\=25000/BINSIZE\={wildcards.resolution}/g' $OUTPUT.ipynb
            jupyter nbconvert --to notebook --execute $OUTPUT.ipynb
            jupyter nbconvert --to html $OUTPUT.nbconvert.ipynb
            rm $OUTPUT.ipynb
        """)