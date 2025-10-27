static void network_to_register(RDMARegister *reg)

{

    reg->key.current_addr = ntohll(reg->key.current_addr);

    reg->current_index = ntohl(reg->current_index);

    reg->chunks = ntohll(reg->chunks);

}
