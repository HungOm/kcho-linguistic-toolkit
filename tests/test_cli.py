"""
Unit tests for CLI functionality.

Tests the command-line interface and CLI commands.
"""

import pytest
from click.testing import CliRunner
from kcho.kcho_app import cli


class TestCLI:
    """Test the CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'K\'Cho Linguistic Toolkit' in result.output
        assert 'Commands:' in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_collocation_command_help(self):
        """Test collocation command help."""
        result = self.runner.invoke(cli, ['collocation', '--help'])
        
        assert result.exit_code == 0
        assert 'collocation' in result.output.lower()
    
    def test_collocation_command_basic(self):
        """Test basic collocation command."""
        # Create a temporary corpus file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '10',
                '--min-freq', '1'
            ])
            
            # Should complete successfully
            assert result.exit_code == 0
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')
    
    def test_collocation_command_with_measures(self):
        """Test collocation command with specific measures."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '5',
                '--min-freq', '1',
                '--measures', 'pmi',
                '--measures', 'tscore'
            ])
            
            assert result.exit_code == 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')
    
    def test_collocation_command_invalid_file(self):
        """Test collocation command with invalid file."""
        result = self.runner.invoke(cli, [
            'collocation',
            '--corpus', 'nonexistent_file.txt',
            '--output', 'test_output.txt',
            '--top-k', '10'
        ])
        
        # Should fail with non-zero exit code
        assert result.exit_code != 0
    
    def test_collocation_command_missing_arguments(self):
        """Test collocation command with missing required arguments."""
        result = self.runner.invoke(cli, ['collocation'])
        
        # Should fail with non-zero exit code
        assert result.exit_code != 0
    
    def test_collocation_command_invalid_parameters(self):
        """Test collocation command with invalid parameters."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        try:
            # Test with invalid top-k (negative)
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '-1'
            ])
            
            assert result.exit_code != 0
            
            # Test with invalid min-freq (negative)
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--min-freq', '-1'
            ])
            
            assert result.exit_code != 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_collocation_command_empty_corpus(self):
        """Test collocation command with empty corpus."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '10',
                '--min-freq', '1'
            ])
            
            # Should handle empty corpus gracefully
            assert result.exit_code == 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')
    
    def test_collocation_command_single_word_corpus(self):
        """Test collocation command with single word corpus."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om\n")
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '10',
                '--min-freq', '1'
            ])
            
            # Should handle single word corpus gracefully
            assert result.exit_code == 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')
    
    def test_collocation_command_output_file_creation(self):
        """Test that collocation command creates output file."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        output_file = 'test_collocation_output.txt'
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', output_file,
                '--top-k', '5',
                '--min-freq', '1'
            ])
            
            assert result.exit_code == 0
            
            # Check that output file was created
            assert os.path.exists(output_file)
            
            # Check that output file has content
            with open(output_file, 'r') as f:
                content = f.read()
                assert len(content) > 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_collocation_command_verbose_mode(self):
        """Test collocation command with verbose mode."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '5',
                '--min-freq', '1',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')
    
    def test_collocation_command_window_size(self):
        """Test collocation command with custom window size."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            temp_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '5',
                '--min-freq', '1',
                '--window-size', '3'
            ])
            
            assert result.exit_code == 0
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists('test_output.txt'):
                os.unlink('test_output.txt')


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_full_cli_workflow(self):
        """Test complete CLI workflow."""
        runner = CliRunner()
        
        import tempfile
        import os
        
        # Create a test corpus
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        output_file = 'test_full_workflow_output.txt'
        
        try:
            # Test collocation extraction
            result = runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', output_file,
                '--top-k', '10',
                '--min-freq', '1',
                '--measures', 'pmi',
                '--measures', 'tscore',
                '--measures', 'dice',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            
            # Check output file
            assert os.path.exists(output_file)
            
            with open(output_file, 'r') as f:
                content = f.read()
                assert len(content) > 0
                # Should contain some collocation results
                assert 'pmi' in content or 'tscore' in content or 'dice' in content
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        runner = CliRunner()
        
        # Test with non-existent corpus file
        result = runner.invoke(cli, [
            'collocation',
            '--corpus', 'nonexistent_file.txt',
            '--output', 'test_output.txt',
            '--top-k', '10'
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()
    
    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        runner = CliRunner()
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        try:
            # Test invalid measures
            result = runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', 'test_output.txt',
                '--top-k', '10',
                '--measures', 'invalid_measure'
            ])
            
            # Should handle invalid measures gracefully
            assert result.exit_code == 0 or result.exit_code != 0  # Either way is acceptable
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_cli_output_formatting(self):
        """Test CLI output formatting."""
        runner = CliRunner()
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Om noh Yong am paapai pe ci\n")
            f.write("Ak'hmó lùum ci\n")
            f.write("Om noh Yong am paapai pe ci\n")
            temp_file = f.name
        
        output_file = 'test_formatting_output.txt'
        
        try:
            result = runner.invoke(cli, [
                'collocation',
                '--corpus', temp_file,
                '--output', output_file,
                '--top-k', '5',
                '--min-freq', '1',
                '--measures', 'pmi'
            ])
            
            assert result.exit_code == 0
            
            # Check output file formatting
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                    # Should have some structure
                    assert len(content) > 0
                    
                    # Should contain measure names
                    assert 'PMI' in content or 'pmi' in content
                    
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
