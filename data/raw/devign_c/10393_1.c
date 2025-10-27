static void network_to_result(RDMARegisterResult *result)

{

    result->rkey = ntohl(result->rkey);

    result->host_addr = ntohll(result->host_addr);

};
