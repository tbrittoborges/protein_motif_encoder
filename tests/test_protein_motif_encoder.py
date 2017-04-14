#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_protein_motif_encoder
----------------------------------

Tests for `protein_motif_encoder` module.
"""
import pandas as pd
import pytest
from click.testing import CliRunner

import cli

def test_command_line_interface():
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'protein_motif_encoder.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
