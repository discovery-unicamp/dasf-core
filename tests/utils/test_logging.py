#!/usr/bin/env python3

import unittest
from unittest.mock import Mock, patch
import logging
import sys

from dasf.utils.logging import init_logging


class TestLogging(unittest.TestCase):
    def test_init_logging_basic(self):
        logger = init_logging()
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "DASF")
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        
        handler = logger.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.stream, sys.stdout)

    def test_init_logging_formatter(self):
        logger = init_logging()
        handler = logger.handlers[0]
        formatter = handler.formatter
        
        self.assertIsInstance(formatter, logging.Formatter)
        self.assertEqual(formatter._fmt, "[%(asctime)s] %(levelname)s - %(message)s")
        self.assertEqual(formatter.datefmt, "%Y-%m-%d %H:%M:%S%z")

    def test_init_logging_clears_existing_handlers(self):
        # First create a logger with handlers
        logger = init_logging()
        self.assertEqual(len(logger.handlers), 1)
        
        # Add another handler manually
        extra_handler = logging.StreamHandler()
        logger.addHandler(extra_handler)
        self.assertEqual(len(logger.handlers), 2)
        
        # Call init_logging again - should clear existing handlers
        logger = init_logging()
        self.assertEqual(len(logger.handlers), 1)

    def test_init_logging_multiple_calls(self):
        # Test that multiple calls return the same logger instance
        logger1 = init_logging()
        logger2 = init_logging()
        
        self.assertIs(logger1, logger2)
        self.assertEqual(len(logger1.handlers), 1)

    @patch('dasf.utils.logging.getLogger')
    def test_init_logging_with_no_existing_handlers(self, mock_get_logger):
        mock_logger = Mock()
        mock_logger.hasHandlers.return_value = False
        mock_get_logger.return_value = mock_logger
        
        init_logging()
        
        mock_logger.setLevel.assert_called_once_with(logging.INFO)
        self.assertTrue(mock_logger.addHandler.called)

    @patch('dasf.utils.logging.getLogger')
    def test_init_logging_with_existing_handlers(self, mock_get_logger):
        mock_logger = Mock()
        mock_logger.hasHandlers.return_value = True
        mock_get_logger.return_value = mock_logger
        
        init_logging()
        
        mock_logger.setLevel.assert_called_once_with(logging.INFO)
        mock_logger.handlers.clear.assert_called_once()
