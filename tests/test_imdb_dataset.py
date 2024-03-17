
import os
from unittest import TestCase

from imdb_dataset import ImdbDataset


class TestImdbDataset(TestCase):
    def test___getitem__(self):
        file = os.path.join(os.path.dirname(__file__),"sample_data", "imdb_original.tsv")
        sut = ImdbDataset(file)
        expected_y = 0
        expected_x = """I had quite high hopes for this film, even though it got a bad review in the paper. I was extremely tolerant, and sat through the entire film. I felt quite sick by the end.<br /><br />Although I am not in the least prude or particularly sensitive to tasteless cinema--I thouroughly enjoyed both Woody Allen's 'Everything You Ever Wanted To Know About Sex,...' and Michael Hanneke's 'Funny Games'--I found the directors' obsession with this ten-year-old wanting to drink women's milk totally sickening. And when the film climaxed in an "orgy" where the boy drinks both his mother's milk, as well as that of the woman he has been lusting after for the whole film, I almost vomited with disgust for the total perversion and sentimental pap that it is.<br /><br />Don't get me wrong, I enjoy the vast majority of European cinema, as well as independently made films, so this flick should have pleased me enormously. Avoid this film at all costs, it should be relegated to the annals of History as a lesson in bad cinema."""

        # Act
        actual_x, actual_y = sut.__getitem__(0)

        # Assert
        self.assertEqual(expected_y, actual_y)
        self.assertEqual(expected_x, actual_x)
