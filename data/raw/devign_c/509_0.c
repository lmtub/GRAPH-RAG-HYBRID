stream_push(StreamSlave *sink, uint8_t *buf, size_t len, uint32_t *app)

{

    StreamSlaveClass *k =  STREAM_SLAVE_GET_CLASS(sink);



    return k->push(sink, buf, len, app);

}
