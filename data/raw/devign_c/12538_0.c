static int curl_aio_flush(void *opaque)

{

    BDRVCURLState *s = opaque;

    int i, j;



    for (i=0; i < CURL_NUM_STATES; i++) {

        for(j=0; j < CURL_NUM_ACB; j++) {

            if (s->states[i].acb[j]) {

                return 1;

            }

        }

    }

    return 0;

}
