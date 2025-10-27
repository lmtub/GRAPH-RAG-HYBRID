static void result_to_network(RDMARegisterResult *result)

{

    result->rkey = htonl(result->rkey);

    result->host_addr = htonll(result->host_addr);

};
