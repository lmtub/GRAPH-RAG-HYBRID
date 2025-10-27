void init_checksum(ByteIOContext *s, unsigned long (*update_checksum)(unsigned long c, const uint8_t *p, unsigned int len), unsigned long checksum){

    s->update_checksum= update_checksum;

    s->checksum= s->update_checksum(checksum, NULL, 0);

    s->checksum_ptr= s->buf_ptr;

}
