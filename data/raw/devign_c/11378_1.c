static void compress_to_network(RDMACompress *comp)

{

    comp->value = htonl(comp->value);

    comp->block_idx = htonl(comp->block_idx);

    comp->offset = htonll(comp->offset);

    comp->length = htonll(comp->length);

}
