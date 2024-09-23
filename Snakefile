### Dendritic cells pipeline
# This snakemake pipeline parallelizes calculations of uni-IPGs to multiple chromosomes (and resolutions, but we'll use only 25 Kb).
# 
# The code takes example notebook executed for chr19 and runs it for all other chromosomes

chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19']#, 'chrX']
resolutions = [25000]

rule all:
    input:
        expand("results/notebook.000.joint-decomp.cis.{chrom}.{resolution}.nbconvert.ipynb", chrom=chroms, resolution=resolutions)

rule test_gpu:
    output:
        file="results/notebook.000.joint-decomp.cis.{chrom}.{resolution}.nbconvert.ipynb",
    run:

        shell(r"""
            # Temporary file for the execution:
            OUTPUT="./results/notebook.000.joint-decomp.cis.{wildcards.chrom}.{wildcards.resolution}"

            # Copy example file to the temporary file:
            cp /home/agalicina/IMMUNE/dendritic_cells/joint-decomp_cis.CHROM.RESOLUTION.ipynb $OUTPUT.ipynb # Make sure to replace with your local path to the notebook

            # Inplace modification of the file (chromosome and bin size):
            sed -i 's/CHROM\=\\\"chr19\\\"/CHROM\=\\\"{wildcards.chrom}\\\"/g' $OUTPUT.ipynb
            sed -i 's/BINSIZE\=25000/BINSIZE\={wildcards.resolution}/g' $OUTPUT.ipynb

            # Inplace execution of the notebook into file with suffix: *.nbconvert.ipynb:
            jupyter nbconvert --to notebook --execute $OUTPUT.ipynb
            jupyter nbconvert --to html $OUTPUT.nbconvert.ipynb

            # Remove temporary file:
            rm $OUTPUT.ipynb
        """)
