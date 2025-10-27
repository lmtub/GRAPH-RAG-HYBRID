static void curl_multi_do(void *arg)

{

    BDRVCURLState *s = (BDRVCURLState *)arg;

    int running;

    int r;



    if (!s->multi) {

        return;

    }



    do {

        r = curl_multi_socket_all(s->multi, &running);

    } while(r == CURLM_CALL_MULTI_PERFORM);



    curl_multi_read(s);

}
