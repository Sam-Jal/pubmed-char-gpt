import unittest

from data.prepare import _extract_abstracts


class PubMedParsingTests(unittest.TestCase):
    def test_structured_abstract_becomes_one_document(self):
        xml = b"""
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <Article>
                <Abstract>
                  <AbstractText Label="BACKGROUND">A long background with an <i>important</i> nested phrase that must remain in the extracted text.</AbstractText>
                  <AbstractText Label="RESULTS">The reported results contain enough additional text to pass the minimum abstract length.</AbstractText>
                </Abstract>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """

        abstracts = _extract_abstracts(xml)

        self.assertEqual(len(abstracts), 1)
        self.assertIn("BACKGROUND:", abstracts[0])
        self.assertIn("important nested phrase", abstracts[0])
        self.assertIn("RESULTS:", abstracts[0])


if __name__ == "__main__":
    unittest.main()
