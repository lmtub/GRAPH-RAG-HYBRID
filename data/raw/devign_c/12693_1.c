static void network_to_compress(RDMACompress *comp)

{

    comp->value = ntohl(comp->value);

    comp->block_idx = ntohl(comp->block_idx);

    comp->offset = ntohll(comp->offset);

    comp->length = ntohll(comp->length);

}
