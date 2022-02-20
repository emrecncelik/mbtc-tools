import requests
from typing import List
from urllib.parse import urljoin
from functools import cached_property
from jpype import JString, JClass, startJVM, shutdownJVM, isJVMStarted, java


class ZemberekDocker:
    def __init__(self, zemberek_url: str = "http://localhost:4567") -> None:
        self.zemberek_url = zemberek_url
        self.header_params = {
            "type": "string",
            "enum": "application/x-www-form-urlencoded; charset=utf-8",
            "default": "application/x-www-form-urlencoded; charset=utf-8",
        }

    def _send_request(self, endpoint: str, data: dict):
        url = urljoin(self.zemberek_url, endpoint)
        response = requests.post(url, headers=self.header_params, data=data)
        response = response.json()
        return response

    def tokenize(self, sentence: str) -> List[str]:
        endpoint = "/simple_tokenization"
        response = self._send_request(endpoint=endpoint, data={"sentence": sentence})
        return response["tokenizations"]

    def detect_sentence_boundaries(self, text: str) -> List[str]:
        endpoint = "/sentence_boundary_detection"
        response = self._send_request(endpoint=endpoint, data={"sentence": text})
        return response["sentences"]

    def pos(self, sentence: str) -> List[str]:
        endpoint = "/find_pos"
        response = self._send_request(endpoint=endpoint, data={"sentence": sentence})
        return [analysis["pos"] for analysis in response]

    def stem(self, token: str) -> str:
        endpoint = "/stems"
        response = self._send_request(endpoint=endpoint, data={"word": token})
        if response["results"]:
            return response["results"][0]["stems"][0]
        else:
            return token

    def lemma(self, token: str) -> str:
        endpoint = "/lemmas"
        response = self._send_request(endpoint=endpoint, data={"word": token})
        if response["results"]:
            return response["results"][0]["lemmas"][0]
        else:
            return token


class ZemberekJava:
    def __init__(
        self,
        zemberek_path: str = "-Djava.class.path=/home/emrecan/workspace/resources/zemberek/bin/zemberek-full.jar",
        zemberek_data_path: str = "/home/emrecan/workspace/resources/zemberek/data",
        java_path: str = "/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so",
    ):
        self.zemberek_path = zemberek_path
        self.zemberek_data_path = zemberek_data_path
        self.java_path = java_path

    def start_jvm(self):
        if not isJVMStarted():
            print("Starting JVM.")
            startJVM(self.java_path, self.zemberek_path, "-ea")
        else:
            print("JVM is already started.")

    def shutdown_jvm(self):
        if isJVMStarted():
            print("Shutting down JVM.")
            shutdownJVM()
        else:
            print("JVM is already down.")

    def tokenize(self, sentence: str) -> List[str]:
        tokens = self._tokenizer.tokenizeToStrings(JString(sentence))
        return [str(token) for token in tokens]

    def detect_sentence_boundaries(self, text: str) -> List[str]:
        sentences = self._extractor.fromParagraph(text)
        return list(sentences)

    def pos(self, sentence: str) -> List[str]:
        analyses: java.util.ArrayList = self._morphology.analyzeAndDisambiguate(
            sentence
        ).bestAnalysis()
        pos = [analysis.getPos().shortForm for analysis in analyses]
        return pos

    def stem(self, token: str) -> str:
        results = self._morphology.analyze(JString(token))
        if results:
            lemma = list(results)[0].getStems()[0]
            return str(lemma)
        else:
            return token

    def lemma(self, token: str):
        results = self._morphology.analyze(JString(token))
        if results:
            lemma = list(results)[0].getLemmas()[0]
            return str(lemma)
        else:
            return token

    def normalize(self, sentence: str) -> str:
        try:
            return str(self._normalizer.normalize(JString(sentence)))
        except java.lang.ArrayIndexOutOfBoundsException:
            print(f"Normalization error with sentence: {sentence}")
            return sentence

    # I don't know if it's a good idea to have a billion cached properties...
    # I don't want to create normalizer, extractor etc. objects multiple times
    # So this is my dumbass solution to the problem
    # Sorry for the cringe, but here we go:

    @cached_property
    def _Paths(self):
        Paths: JClass = JClass("java.nio.file.Paths")
        return Paths

    @cached_property
    def _WordAnalysis(self):
        WordAnalysis: JClass = JClass("zemberek.morphology.analysis.WordAnalysis")
        return WordAnalysis

    @cached_property
    def _TurkishMorphology(self):
        TurkishMorphology: JClass = JClass("zemberek.morphology.TurkishMorphology")
        return TurkishMorphology

    @cached_property
    def _TurkishSentenceNormalizer(self):
        TurkishSentenceNormalizer: JClass = JClass(
            "zemberek.normalization.TurkishSentenceNormalizer"
        )
        return TurkishSentenceNormalizer

    @cached_property
    def _TurkishSentenceExtractor(self):
        TurkishSentenceExtractor: JClass = JClass(
            "zemberek.tokenization.TurkishSentenceExtractor"
        )
        return TurkishSentenceExtractor

    @cached_property
    def _TurkishTokenizer(self):
        TurkishTokenizer: JClass = JClass("zemberek.tokenization.TurkishTokenizer")
        return TurkishTokenizer

    @cached_property
    def _morphology(self):
        morphology: self._TurkishMorphology = (
            self._TurkishMorphology.createWithDefaults()
        )
        return morphology

    @cached_property
    def _normalizer(self):
        normalizer = self._TurkishSentenceNormalizer(
            self._morphology,
            self._Paths.get(self.zemberek_data_path + "/normalization"),
            self._Paths.get(self.zemberek_data_path + "/lm/lm.2gram.slm"),
        )
        return normalizer

    @cached_property
    def _extractor(self):
        extractor: self._TurkishSentenceExtractor = (
            self._TurkishSentenceExtractor.DEFAULT
        )
        return extractor

    @cached_property
    def _tokenizer(self):
        tokenizer: self._TurkishTokenizer = self._TurkishTokenizer.DEFAULT
        return tokenizer
