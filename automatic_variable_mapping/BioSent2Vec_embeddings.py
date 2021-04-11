from gensim.models import KeyedVectors


def convert_to_keyedvectors(input_file_name, output_file_name):
    model = KeyedVectors.load_word2vec_format(input_file_name,
                                              binary=True,
                                              limit=None)
    model.save_word2vec_format(output_file_name,
                               binary=False)


def load_keyed_vectors(intup_file_name):
    return KeyedVectors.load(dir + "bio-word-vectors.vec", mmap='r')


working_dir = "/run/media/harrison/external-hd/data/"
dir = working_dir + "WordVectors2/"

binary_file_name = dir + "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
vector_file_name = dir + "bio-word-vectors.vec"

convert_to_keyedvectors(binary_file_name, vector_file_name)

model = load_keyed_vectors(vector_file_name)
