import sqlite3
import numpy as np
import database_numpy_modifications as dnm
import io

# Converts TEXT to np.array when selecting
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle = True)

def get_descriptor_list():
    sqlite3.register_adapter(np.ndarray, dnm.adapt_array)
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect('ImSeDatabaseORBCLUSTER.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()

    c.execute("""SELECT orbDescriptors FROM imgFeatures""")

    descriptor_list = []

    for row in c.fetchall():
        descriptor_list.append(row[0])

    descriptors = descriptor_list[0]

    # These to protect us against two errors that would occur otherwise. Or one at least.
    first_it = True
    i = 0

    # this number is actually rather low because otherwise it would be very slow
    number_of_reference_pictures = 400

    for descriptor in descriptor_list:
        if first_it == False:
            descriptors = np.vstack((descriptors, descriptor))
            i = i + 5
        else:
            first_it = False
        if i == number_of_reference_pictures:
            break
        else:
            pass

    # Adapt form of descriptors
    descriptors.shape
    descriptors_float = descriptors.astype(float)

    return descriptors_float


get_descriptor_list()