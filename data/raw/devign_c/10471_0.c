static int index_search_timestamp(AVIndexEntry *entries, 

                                  int nb_entries, int wanted_timestamp)

{

    int a, b, m;

    int64_t timestamp;



    if (nb_entries <= 0)

        return -1;

    

    a = 0;

    b = nb_entries - 1;

    while (a <= b) {

        m = (a + b) >> 1;

        timestamp = entries[m].timestamp;

        if (timestamp == wanted_timestamp)

            goto found;

        else if (timestamp > wanted_timestamp) {

            b = m - 1;

        } else {

            a = m + 1;

        }

    }

    m = a;

    if (m > 0)

        m--;

 found:

    return m;

}
