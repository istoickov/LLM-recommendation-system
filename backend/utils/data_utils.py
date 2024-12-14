import re

from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from consts.text_replacements import EMOJIS
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

download("stopwords")
download("punkt")
download("wordnet")
download("punkt_tab")
download("punkt")
download("wordnet")
download("omw-1.4")

stop_words = list(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class DataUtils:
    def make_basic_summary(self, person_data, instagram_data):
        name = person_data.get("name", "Unknown")
        location = person_data.get("state", "Unknown")
        tags = ", ".join([tag.get("value") for tag in person_data.get("tags", [])])

        full_name = instagram_data.get("full_name", "")
        bio = instagram_data.get("bio", "")
        followers = instagram_data.get("follows", 0)
        following = instagram_data.get("following", 0)
        post_count = instagram_data.get("timeline_count", 0)

        summary = (
            f"Profile of {name} based in {location}. "
            f"Tagged under: {tags}. "
            f"Instagram profile name: {full_name}. Bio: {bio}. "
            f"Follows {followers} accounts and is followed by {following} accounts. "
            f"Total posts on Instagram: {post_count}. "
        )
        return summary

    def make_post_summary(self, post):
        caption = post.get("caption", "")
        likes = post.get("liked_count", 0)
        views = post.get("viewed_count", 0)
        comments = post.get("comment_count", 0)
        post_summary = f"Post captioned '{caption}' received {likes} likes, {views} views, and {comments} comments."
        return post_summary

    def make_summaries(self, data):
        summaries = []
        for person_data in data:
            instagram_data = person_data.get("instagram", {})
            post_summaries = []
            for post in instagram_data.get("posts", []):
                post_summary = self.make_post_summary(post)
                post_summaries.append(post_summary)

            summary = self.make_basic_summary(person_data, instagram_data)
            summary += " ".join(post_summaries)
            summaries.append(summary)

        return summaries

    def clean_summary(self, summary):
        # Convert text to lowercase
        cleaned_summary = summary.lower()
        # Remove extra spaces and line breaks
        cleaned_summary = re.sub(r"\s+", " ", cleaned_summary).strip()
        # Remove emojis
        cleaned_summary = re.sub(r'[^\w\s,.:;!?\'"]+', "", cleaned_summary)
        # Replace "None" with "Unknown"
        cleaned_summary = cleaned_summary.replace("none", "unknown")
        # Remove URLs
        cleaned_summary = re.sub(r"http\S+|www\S+", "", cleaned_summary)
        # Remove hashtags and @mentions if not needed
        cleaned_summary = re.sub(r"#\S+", "", cleaned_summary)  # Remove hashtags
        cleaned_summary = re.sub(r"@\S+", "", cleaned_summary)  # Remove mentions
        # Fix punctuation (remove redundant commas or periods)
        cleaned_summary = re.sub(r"([.,!?])\1+", r"\1", cleaned_summary)

        return cleaned_summary

    def clean_summaries(self, data):
        cleaned_summaries = []
        for summary in data:
            cleaned_summary = self.clean_summary(summary)
            cleaned_summaries.append(cleaned_summary)

        return cleaned_summaries

    def stemm_sentence(self, sentence):
        tokens = word_tokenize(sentence.lower())
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return stemmed_tokens

    def stemm_data(self, data):
        new_data = []
        for text in data:
            stemmed_tokens = self.stemm_sentence(text)
            new_data.append(" ".join(stemmed_tokens))
        return new_data

    def lemmatization_senetence(self, senetence):
        tokens = word_tokenize(senetence.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def lemmatization_data(self, data):
        new_data = []
        for text in data:
            lemmatized_tokens = self.lemmatization_senetence(text)
            new_data.append(" ".join(lemmatized_tokens))
        return new_data

    def remove_stopwords(self, data):
        new_data = []
        for item in data:
            text = " ".join([word for word in item.split() if word not in stop_words])
            new_data.append(text)

        return new_data
