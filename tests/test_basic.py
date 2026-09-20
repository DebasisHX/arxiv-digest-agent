import unittest

from app.nodes.query import detect_query_type
from app.nodes.chunking import create_chunks


class TestQueryUnderstanding(unittest.TestCase):

    def test_arxiv_id(self):
        self.assertEqual(
            detect_query_type("1706.03762"),
            "paper"
        )

    def test_arxiv_url(self):
        self.assertEqual(
            detect_query_type("https://arxiv.org/abs/1706.03762"),
            "paper"
        )

    def test_topic_query(self):
        self.assertEqual(
            detect_query_type("transformer architecture"),
            "topic"
        )


class TestChunking(unittest.TestCase):

    def test_chunk_creation(self):
        text = "A" * 3000

        chunks = create_chunks(
            text,
            chunk_size=1000,
            chunk_overlap=100
        )

        self.assertGreater(len(chunks), 1)

    def test_empty_text(self):
        chunks = create_chunks("")

        self.assertEqual(chunks, [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            create_chunks(
                "some text",
                chunk_size=100,
                chunk_overlap=100
            )


if __name__ == "__main__":
    unittest.main()