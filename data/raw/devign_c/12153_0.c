static void rtl8139_do_receive(void *opaque, const uint8_t *buf, int size, int do_interrupt)

{

    RTL8139State *s = opaque;



    uint32_t packet_header = 0;



    uint8_t buf1[60];

    static const uint8_t broadcast_macaddr[6] = 

        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };



    DEBUG_PRINT((">>> RTL8139: received len=%d\n", size));



    /* test if board clock is stopped */

    if (!s->clock_enabled)

    {

        DEBUG_PRINT(("RTL8139: stopped ==========================\n"));

        return;

    }



    /* first check if receiver is enabled */



    if (!rtl8139_receiver_enabled(s))

    {

        DEBUG_PRINT(("RTL8139: receiver disabled ================\n"));

        return;

    }



    /* XXX: check this */

    if (s->RxConfig & AcceptAllPhys) {

        /* promiscuous: receive all */

        DEBUG_PRINT((">>> RTL8139: packet received in promiscuous mode\n"));



    } else {

        if (!memcmp(buf,  broadcast_macaddr, 6)) {

            /* broadcast address */

            if (!(s->RxConfig & AcceptBroadcast))

            {

                DEBUG_PRINT((">>> RTL8139: broadcast packet rejected\n"));



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return;

            }



            packet_header |= RxBroadcast;



            DEBUG_PRINT((">>> RTL8139: broadcast packet received\n"));



            /* update tally counter */

            ++s->tally_counters.RxOkBrd;



        } else if (buf[0] & 0x01) {

            /* multicast */

            if (!(s->RxConfig & AcceptMulticast))

            {

                DEBUG_PRINT((">>> RTL8139: multicast packet rejected\n"));



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return;

            }



            int mcast_idx = compute_mcast_idx(buf);



            if (!(s->mult[mcast_idx >> 3] & (1 << (mcast_idx & 7))))

            {

                DEBUG_PRINT((">>> RTL8139: multicast address mismatch\n"));



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return;

            }



            packet_header |= RxMulticast;



            DEBUG_PRINT((">>> RTL8139: multicast packet received\n"));



            /* update tally counter */

            ++s->tally_counters.RxOkMul;



        } else if (s->phys[0] == buf[0] &&

                   s->phys[1] == buf[1] &&                   

                   s->phys[2] == buf[2] &&            

                   s->phys[3] == buf[3] &&            

                   s->phys[4] == buf[4] &&            

                   s->phys[5] == buf[5]) {

            /* match */

            if (!(s->RxConfig & AcceptMyPhys))

            {

                DEBUG_PRINT((">>> RTL8139: rejecting physical address matching packet\n"));



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return;

            }



            packet_header |= RxPhysical;



            DEBUG_PRINT((">>> RTL8139: physical address matching packet received\n"));



            /* update tally counter */

            ++s->tally_counters.RxOkPhy;



        } else {



            DEBUG_PRINT((">>> RTL8139: unknown packet\n"));



            /* update tally counter */

            ++s->tally_counters.RxERR;



            return;

        }

    }



    /* if too small buffer, then expand it */

    if (size < MIN_BUF_SIZE) {

        memcpy(buf1, buf, size);

        memset(buf1 + size, 0, MIN_BUF_SIZE - size);

        buf = buf1;

        size = MIN_BUF_SIZE;

    }



    if (rtl8139_cp_receiver_enabled(s))

    {

        DEBUG_PRINT(("RTL8139: in C+ Rx mode ================\n"));



        /* begin C+ receiver mode */



/* w0 ownership flag */

#define CP_RX_OWN (1<<31)

/* w0 end of ring flag */

#define CP_RX_EOR (1<<30)

/* w0 bits 0...12 : buffer size */

#define CP_RX_BUFFER_SIZE_MASK ((1<<13) - 1)

/* w1 tag available flag */

#define CP_RX_TAVA (1<<16)

/* w1 bits 0...15 : VLAN tag */

#define CP_RX_VLAN_TAG_MASK ((1<<16) - 1)

/* w2 low  32bit of Rx buffer ptr */

/* w3 high 32bit of Rx buffer ptr */



        int descriptor = s->currCPlusRxDesc;

        target_phys_addr_t cplus_rx_ring_desc;



        cplus_rx_ring_desc = rtl8139_addr64(s->RxRingAddrLO, s->RxRingAddrHI);

        cplus_rx_ring_desc += 16 * descriptor;



        DEBUG_PRINT(("RTL8139: +++ C+ mode reading RX descriptor %d from host memory at %08x %08x = %016" PRIx64 "\n",

               descriptor, s->RxRingAddrHI, s->RxRingAddrLO, (uint64_t)cplus_rx_ring_desc));



        uint32_t val, rxdw0,rxdw1,rxbufLO,rxbufHI;



        cpu_physical_memory_read(cplus_rx_ring_desc,    (uint8_t *)&val, 4);

        rxdw0 = le32_to_cpu(val);

        cpu_physical_memory_read(cplus_rx_ring_desc+4,  (uint8_t *)&val, 4);

        rxdw1 = le32_to_cpu(val);

        cpu_physical_memory_read(cplus_rx_ring_desc+8,  (uint8_t *)&val, 4);

        rxbufLO = le32_to_cpu(val);

        cpu_physical_memory_read(cplus_rx_ring_desc+12, (uint8_t *)&val, 4);

        rxbufHI = le32_to_cpu(val);



        DEBUG_PRINT(("RTL8139: +++ C+ mode RX descriptor %d %08x %08x %08x %08x\n",

               descriptor,

               rxdw0, rxdw1, rxbufLO, rxbufHI));



        if (!(rxdw0 & CP_RX_OWN))

        {

            DEBUG_PRINT(("RTL8139: C+ Rx mode : descriptor %d is owned by host\n", descriptor));



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;



            /* update tally counter */

            ++s->tally_counters.RxERR;

            ++s->tally_counters.MissPkt;



            rtl8139_update_irq(s);

            return;

        }



        uint32_t rx_space = rxdw0 & CP_RX_BUFFER_SIZE_MASK;



        /* TODO: scatter the packet over available receive ring descriptors space */



        if (size+4 > rx_space)

        {

            DEBUG_PRINT(("RTL8139: C+ Rx mode : descriptor %d size %d received %d + 4\n",

                   descriptor, rx_space, size));



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;



            /* update tally counter */

            ++s->tally_counters.RxERR;

            ++s->tally_counters.MissPkt;



            rtl8139_update_irq(s);

            return;

        }



        target_phys_addr_t rx_addr = rtl8139_addr64(rxbufLO, rxbufHI);



        /* receive/copy to target memory */

        cpu_physical_memory_write( rx_addr, buf, size );



        if (s->CpCmd & CPlusRxChkSum)

        {

            /* do some packet checksumming */

        }



        /* write checksum */

#if defined (RTL8139_CALCULATE_RXCRC)

        val = cpu_to_le32(crc32(~0, buf, size));

#else

        val = 0;

#endif

        cpu_physical_memory_write( rx_addr+size, (uint8_t *)&val, 4);



/* first segment of received packet flag */

#define CP_RX_STATUS_FS (1<<29)

/* last segment of received packet flag */

#define CP_RX_STATUS_LS (1<<28)

/* multicast packet flag */

#define CP_RX_STATUS_MAR (1<<26)

/* physical-matching packet flag */

#define CP_RX_STATUS_PAM (1<<25)

/* broadcast packet flag */

#define CP_RX_STATUS_BAR (1<<24)

/* runt packet flag */

#define CP_RX_STATUS_RUNT (1<<19)

/* crc error flag */

#define CP_RX_STATUS_CRC (1<<18)

/* IP checksum error flag */

#define CP_RX_STATUS_IPF (1<<15)

/* UDP checksum error flag */

#define CP_RX_STATUS_UDPF (1<<14)

/* TCP checksum error flag */

#define CP_RX_STATUS_TCPF (1<<13)



        /* transfer ownership to target */

        rxdw0 &= ~CP_RX_OWN;



        /* set first segment bit */

        rxdw0 |= CP_RX_STATUS_FS;



        /* set last segment bit */

        rxdw0 |= CP_RX_STATUS_LS;



        /* set received packet type flags */

        if (packet_header & RxBroadcast)

            rxdw0 |= CP_RX_STATUS_BAR;

        if (packet_header & RxMulticast)

            rxdw0 |= CP_RX_STATUS_MAR;

        if (packet_header & RxPhysical)

            rxdw0 |= CP_RX_STATUS_PAM;



        /* set received size */

        rxdw0 &= ~CP_RX_BUFFER_SIZE_MASK;

        rxdw0 |= (size+4);



        /* reset VLAN tag flag */

        rxdw1 &= ~CP_RX_TAVA;



        /* update ring data */

        val = cpu_to_le32(rxdw0);

        cpu_physical_memory_write(cplus_rx_ring_desc,    (uint8_t *)&val, 4);

        val = cpu_to_le32(rxdw1);

        cpu_physical_memory_write(cplus_rx_ring_desc+4,  (uint8_t *)&val, 4);



        /* update tally counter */

        ++s->tally_counters.RxOk;



        /* seek to next Rx descriptor */

        if (rxdw0 & CP_RX_EOR)

        {

            s->currCPlusRxDesc = 0;

        }

        else

        {

            ++s->currCPlusRxDesc;

        }



        DEBUG_PRINT(("RTL8139: done C+ Rx mode ----------------\n"));



    }

    else

    {

        DEBUG_PRINT(("RTL8139: in ring Rx mode ================\n"));



        /* begin ring receiver mode */

        int avail = MOD2(s->RxBufferSize + s->RxBufPtr - s->RxBufAddr, s->RxBufferSize);



        /* if receiver buffer is empty then avail == 0 */



        if (avail != 0 && size + 8 >= avail)

        {

            DEBUG_PRINT(("rx overflow: rx buffer length %d head 0x%04x read 0x%04x === available 0x%04x need 0x%04x\n",

                   s->RxBufferSize, s->RxBufAddr, s->RxBufPtr, avail, size + 8));



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;

            rtl8139_update_irq(s);

            return;

        }



        packet_header |= RxStatusOK;



        packet_header |= (((size+4) << 16) & 0xffff0000);



        /* write header */

        uint32_t val = cpu_to_le32(packet_header);



        rtl8139_write_buffer(s, (uint8_t *)&val, 4);



        rtl8139_write_buffer(s, buf, size);



        /* write checksum */

#if defined (RTL8139_CALCULATE_RXCRC)

        val = cpu_to_le32(crc32(~0, buf, size));

#else

        val = 0;

#endif



        rtl8139_write_buffer(s, (uint8_t *)&val, 4);



        /* correct buffer write pointer */

        s->RxBufAddr = MOD2((s->RxBufAddr + 3) & ~0x3, s->RxBufferSize);



        /* now we can signal we have received something */



        DEBUG_PRINT(("   received: rx buffer length %d head 0x%04x read 0x%04x\n",

               s->RxBufferSize, s->RxBufAddr, s->RxBufPtr));

    }



    s->IntrStatus |= RxOK;



    if (do_interrupt)

    {

        rtl8139_update_irq(s);

    }

}
