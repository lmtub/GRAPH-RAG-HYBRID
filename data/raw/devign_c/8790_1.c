bool arp_table_search(Slirp *slirp, uint32_t ip_addr,

                      uint8_t out_ethaddr[ETH_ALEN])

{

    const uint32_t broadcast_addr =

        ~slirp->vnetwork_mask.s_addr | slirp->vnetwork_addr.s_addr;

    ArpTable *arptbl = &slirp->arp_table;

    int i;



    DEBUG_CALL("arp_table_search");

    DEBUG_ARG("ip = 0x%x", ip_addr);



    /* Check 0.0.0.0/8 invalid source-only addresses */

    assert((ip_addr & htonl(~(0xf << 28))) != 0);



    /* If broadcast address */

    if (ip_addr == 0xffffffff || ip_addr == broadcast_addr) {

        /* return Ethernet broadcast address */

        memset(out_ethaddr, 0xff, ETH_ALEN);

        return 1;

    }



    for (i = 0; i < ARP_TABLE_SIZE; i++) {

        if (arptbl->table[i].ar_sip == ip_addr) {

            memcpy(out_ethaddr, arptbl->table[i].ar_sha,  ETH_ALEN);

            DEBUG_ARGS((dfd, " found hw addr = %02x:%02x:%02x:%02x:%02x:%02x\n",

                        out_ethaddr[0], out_ethaddr[1], out_ethaddr[2],

                        out_ethaddr[3], out_ethaddr[4], out_ethaddr[5]));

            return 1;

        }

    }



    return 0;

}
