static void remote_block_to_network(RDMARemoteBlock *rb)

{

    rb->remote_host_addr = htonll(rb->remote_host_addr);

    rb->offset = htonll(rb->offset);

    rb->length = htonll(rb->length);

    rb->remote_rkey = htonl(rb->remote_rkey);

}
