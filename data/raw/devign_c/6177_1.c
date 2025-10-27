static ssize_t gem_receive(VLANClientState *nc, const uint8_t *buf, size_t size)

{

    unsigned    desc[2];

    target_phys_addr_t packet_desc_addr, last_desc_addr;

    GemState *s;

    unsigned   rxbufsize, bytes_to_copy;

    unsigned   rxbuf_offset;

    uint8_t    rxbuf[2048];

    uint8_t   *rxbuf_ptr;



    s = DO_UPCAST(NICState, nc, nc)->opaque;



    /* Do nothing if receive is not enabled. */

    if (!(s->regs[GEM_NWCTRL] & GEM_NWCTRL_RXENA)) {

        return -1;

    }



    /* Is this destination MAC address "for us" ? */

    if (gem_mac_address_filter(s, buf) == GEM_RX_REJECT) {

        return -1;

    }



    /* Discard packets with receive length error enabled ? */

    if (s->regs[GEM_NWCFG] & GEM_NWCFG_LERR_DISC) {

        unsigned type_len;



        /* Fish the ethertype / length field out of the RX packet */

        type_len = buf[12] << 8 | buf[13];

        /* It is a length field, not an ethertype */

        if (type_len < 0x600) {

            if (size < type_len) {

                /* discard */

                return -1;

            }

        }

    }



    /*

     * Determine configured receive buffer offset (probably 0)

     */

    rxbuf_offset = (s->regs[GEM_NWCFG] & GEM_NWCFG_BUFF_OFST_M) >>

                   GEM_NWCFG_BUFF_OFST_S;



    /* The configure size of each receive buffer.  Determines how many

     * buffers needed to hold this packet.

     */

    rxbufsize = ((s->regs[GEM_DMACFG] & GEM_DMACFG_RBUFSZ_M) >>

                 GEM_DMACFG_RBUFSZ_S) * GEM_DMACFG_RBUFSZ_MUL;

    bytes_to_copy = size;



    /* Strip of FCS field ? (usually yes) */

    if (s->regs[GEM_NWCFG] & GEM_NWCFG_STRIP_FCS) {

        rxbuf_ptr = (void *)buf;

    } else {

        unsigned crc_val;

        int      crc_offset;



        /* The application wants the FCS field, which QEMU does not provide.

         * We must try and caclculate one.

         */



        memcpy(rxbuf, buf, size);

        memset(rxbuf + size, 0, sizeof(rxbuf - size));

        rxbuf_ptr = rxbuf;

        crc_val = cpu_to_le32(crc32(0, rxbuf, MAX(size, 60)));

        if (size < 60) {

            crc_offset = 60;

        } else {

            crc_offset = size;

        }

        memcpy(rxbuf + crc_offset, &crc_val, sizeof(crc_val));



        bytes_to_copy += 4;

        size += 4;

    }



    /* Pad to minimum length */

    if (size < 64) {

        size = 64;

    }



    DB_PRINT("config bufsize: %d packet size: %ld\n", rxbufsize, size);



    packet_desc_addr = s->rx_desc_addr;

    while (1) {

        DB_PRINT("read descriptor 0x%x\n", packet_desc_addr);

        /* read current descriptor */

        cpu_physical_memory_read(packet_desc_addr,

                                 (uint8_t *)&desc[0], sizeof(desc));



        /* Descriptor owned by software ? */

        if (rx_desc_get_ownership(desc) == 1) {

            DB_PRINT("descriptor 0x%x owned by sw.\n", packet_desc_addr);

            s->regs[GEM_RXSTATUS] |= GEM_RXSTATUS_NOBUF;

            /* Handle interrupt consequences */

            gem_update_int_status(s);

            return -1;

        }



        DB_PRINT("copy %d bytes to 0x%x\n", MIN(bytes_to_copy, rxbufsize),

                rx_desc_get_buffer(desc));



        /*

         * Let's have QEMU lend a helping hand.

         */

        if (rx_desc_get_buffer(desc) == 0) {

            DB_PRINT("Invalid RX buffer (NULL) for descriptor 0x%x\n",

                       packet_desc_addr);

            break;

        }



        /* Copy packet data to emulated DMA buffer */

        cpu_physical_memory_write(rx_desc_get_buffer(desc) + rxbuf_offset,

                                  rxbuf_ptr, MIN(bytes_to_copy, rxbufsize));

        bytes_to_copy -= MIN(bytes_to_copy, rxbufsize);

        rxbuf_ptr += MIN(bytes_to_copy, rxbufsize);

        if (bytes_to_copy == 0) {

            break;

        }



        /* Next descriptor */

        if (rx_desc_get_wrap(desc)) {

            packet_desc_addr = s->regs[GEM_RXQBASE];

        } else {

            packet_desc_addr += 8;

        }

    }



    DB_PRINT("set length: %ld, EOF on descriptor 0x%x\n", size,

            (unsigned)packet_desc_addr);



    /* Update last descriptor with EOF and total length */

    rx_desc_set_eof(desc);

    rx_desc_set_length(desc, size);

    cpu_physical_memory_write(packet_desc_addr,

                              (uint8_t *)&desc[0], sizeof(desc));



    /* Advance RX packet descriptor Q */

    last_desc_addr = packet_desc_addr;

    packet_desc_addr = s->rx_desc_addr;

    s->rx_desc_addr = last_desc_addr;

    if (rx_desc_get_wrap(desc)) {

        s->rx_desc_addr = s->regs[GEM_RXQBASE];

    } else {

        s->rx_desc_addr += 8;

    }



    DB_PRINT("set SOF, OWN on descriptor 0x%08x\n", packet_desc_addr);



    /* Count it */

    gem_receive_updatestats(s, buf, size);



    /* Update first descriptor (which could also be the last) */

    /* read descriptor */

    cpu_physical_memory_read(packet_desc_addr,

                             (uint8_t *)&desc[0], sizeof(desc));

    rx_desc_set_sof(desc);

    rx_desc_set_ownership(desc);

    cpu_physical_memory_write(packet_desc_addr,

                              (uint8_t *)&desc[0], sizeof(desc));



    s->regs[GEM_RXSTATUS] |= GEM_RXSTATUS_FRMRCVD;



    /* Handle interrupt consequences */

    gem_update_int_status(s);



    return size;

}
