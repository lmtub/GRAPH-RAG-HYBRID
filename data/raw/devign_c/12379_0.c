static ssize_t eth_rx(NetClientState *nc, const uint8_t *buf, size_t size)

{

    XilinxAXIEnet *s = qemu_get_nic_opaque(nc);

    static const unsigned char sa_bcast[6] = {0xff, 0xff, 0xff,

                                              0xff, 0xff, 0xff};

    static const unsigned char sa_ipmcast[3] = {0x01, 0x00, 0x52};

    uint32_t app[6] = {0};

    int promisc = s->fmi & (1 << 31);

    int unicast, broadcast, multicast, ip_multicast = 0;

    uint32_t csum32;

    uint16_t csum16;

    int i;



    DENET(qemu_log("%s: %zd bytes\n", __func__, size));



    unicast = ~buf[0] & 0x1;

    broadcast = memcmp(buf, sa_bcast, 6) == 0;

    multicast = !unicast && !broadcast;

    if (multicast && (memcmp(sa_ipmcast, buf, sizeof sa_ipmcast) == 0)) {

        ip_multicast = 1;

    }



    /* Jumbo or vlan sizes ?  */

    if (!(s->rcw[1] & RCW1_JUM)) {

        if (size > 1518 && size <= 1522 && !(s->rcw[1] & RCW1_VLAN)) {

            return size;

        }

    }



    /* Basic Address filters.  If you want to use the extended filters

       you'll generally have to place the ethernet mac into promiscuous mode

       to avoid the basic filtering from dropping most frames.  */

    if (!promisc) {

        if (unicast) {

            if (!enet_match_addr(buf, s->uaw[0], s->uaw[1])) {

                return size;

            }

        } else {

            if (broadcast) {

                /* Broadcast.  */

                if (s->regs[R_RAF] & RAF_BCAST_REJ) {

                    return size;

                }

            } else {

                int drop = 1;



                /* Multicast.  */

                if (s->regs[R_RAF] & RAF_MCAST_REJ) {

                    return size;

                }



                for (i = 0; i < 4; i++) {

                    if (enet_match_addr(buf, s->maddr[i][0], s->maddr[i][1])) {

                        drop = 0;

                        break;

                    }

                }



                if (drop) {

                    return size;

                }

            }

        }

    }



    /* Extended mcast filtering enabled?  */

    if (axienet_newfunc_enabled(s) && axienet_extmcf_enabled(s)) {

        if (unicast) {

            if (!enet_match_addr(buf, s->ext_uaw[0], s->ext_uaw[1])) {

                return size;

            }

        } else {

            if (broadcast) {

                /* Broadcast. ???  */

                if (s->regs[R_RAF] & RAF_BCAST_REJ) {

                    return size;

                }

            } else {

                int idx, bit;



                /* Multicast.  */

                if (!memcmp(buf, sa_ipmcast, 3)) {

                    return size;

                }



                idx  = (buf[4] & 0x7f) << 8;

                idx |= buf[5];



                bit = 1 << (idx & 0x1f);

                idx >>= 5;



                if (!(s->ext_mtable[idx] & bit)) {

                    return size;

                }

            }

        }

    }



    if (size < 12) {

        s->regs[R_IS] |= IS_RX_REJECT;

        enet_update_irq(s);

        return -1;

    }



    if (size > (s->c_rxmem - 4)) {

        size = s->c_rxmem - 4;

    }



    memcpy(s->rxmem, buf, size);

    memset(s->rxmem + size, 0, 4); /* Clear the FCS.  */



    if (s->rcw[1] & RCW1_FCS) {

        size += 4; /* fcs is inband.  */

    }



    app[0] = 5 << 28;

    csum32 = net_checksum_add(size - 14, (uint8_t *)s->rxmem + 14);

    /* Fold it once.  */

    csum32 = (csum32 & 0xffff) + (csum32 >> 16);

    /* And twice to get rid of possible carries.  */

    csum16 = (csum32 & 0xffff) + (csum32 >> 16);

    app[3] = csum16;

    app[4] = size & 0xffff;



    s->stats.rx_bytes += size;

    s->stats.rx++;

    if (multicast) {

        s->stats.rx_mcast++;

        app[2] |= 1 | (ip_multicast << 1);

    } else if (broadcast) {

        s->stats.rx_bcast++;

        app[2] |= 1 << 3;

    }



    /* Good frame.  */

    app[2] |= 1 << 6;



    s->rxsize = size;

    s->rxpos = 0;

    s->rxapp = g_memdup(app, sizeof(app));

    axienet_eth_rx_notify(s);



    enet_update_irq(s);

    return size;

}
