# -*- coding: utf-8 -*-

import click


@click.command()
def main(args=None):
    """Console script for protein_motif_encoder"""
    click.echo("Set of encoding strategies for protein motifs sequences training with machine "
               "learning. Motif most have the same number of amino acids."
               "protein_motif_encoder.cli.main")
    # TODO


if __name__ == "__main__":
    main()
