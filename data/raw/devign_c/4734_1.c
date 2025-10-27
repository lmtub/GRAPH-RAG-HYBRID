static int compute_send_delay(HTTPContext *c)

{

    int datarate = 8 * get_longterm_datarate(&c->datarate, c->data_count); 



    if (datarate > c->stream->bandwidth * 2000) {

        return 1000;

    }

    return 0;

}
