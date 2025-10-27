static BOOTPClient *get_new_addr(struct in_addr *paddr)

{

    BOOTPClient *bc;

    int i;



    for(i = 0; i < NB_ADDR; i++) {

        if (!bootp_clients[i].allocated)

            goto found;

    }

    return NULL;

 found:

    bc = &bootp_clients[i];

    bc->allocated = 1;

    paddr->s_addr = htonl(ntohl(special_addr.s_addr) | (i + START_ADDR));

    return bc;

}
