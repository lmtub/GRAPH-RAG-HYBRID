e1000_receive(VLANClientState *nc, const uint8_t *buf, size_t size)

{

    E1000State *s = DO_UPCAST(NICState, nc, nc)->opaque;

    struct e1000_rx_desc desc;

    target_phys_addr_t base;

    unsigned int n, rdt;

    uint32_t rdh_start;

    uint16_t vlan_special = 0;

    uint8_t vlan_status = 0, vlan_offset = 0;

    uint8_t min_buf[MIN_BUF_SIZE];

    size_t desc_offset;

    size_t desc_size;

    size_t total_size;



    if (!(s->mac_reg[RCTL] & E1000_RCTL_EN))

        return -1;



    /* Pad to minimum Ethernet frame length */

    if (size < sizeof(min_buf)) {

        memcpy(min_buf, buf, size);

        memset(&min_buf[size], 0, sizeof(min_buf) - size);

        buf = min_buf;

        size = sizeof(min_buf);

    }



    if (!receive_filter(s, buf, size))

        return size;



    if (vlan_enabled(s) && is_vlan_packet(s, buf)) {

        vlan_special = cpu_to_le16(be16_to_cpup((uint16_t *)(buf + 14)));

        memmove((uint8_t *)buf + 4, buf, 12);

        vlan_status = E1000_RXD_STAT_VP;

        vlan_offset = 4;

        size -= 4;

    }



    rdh_start = s->mac_reg[RDH];

    desc_offset = 0;

    total_size = size + fcs_len(s);

    do {

        desc_size = total_size - desc_offset;

        if (desc_size > s->rxbuf_size) {

            desc_size = s->rxbuf_size;

        }

        if (s->mac_reg[RDH] == s->mac_reg[RDT] && s->check_rxov) {

            /* Discard all data written so far */

            s->mac_reg[RDH] = rdh_start;

            set_ics(s, 0, E1000_ICS_RXO);

            return -1;

        }

        base = ((uint64_t)s->mac_reg[RDBAH] << 32) + s->mac_reg[RDBAL] +

               sizeof(desc) * s->mac_reg[RDH];

        cpu_physical_memory_read(base, (void *)&desc, sizeof(desc));

        desc.special = vlan_special;

        desc.status |= (vlan_status | E1000_RXD_STAT_DD);

        if (desc.buffer_addr) {

            if (desc_offset < size) {

                size_t copy_size = size - desc_offset;

                if (copy_size > s->rxbuf_size) {

                    copy_size = s->rxbuf_size;

                }

                cpu_physical_memory_write(le64_to_cpu(desc.buffer_addr),

                                          (void *)(buf + desc_offset + vlan_offset),

                                          copy_size);

            }

            desc_offset += desc_size;

            desc.length = cpu_to_le16(desc_size);

            if (desc_offset >= total_size) {

                desc.status |= E1000_RXD_STAT_EOP | E1000_RXD_STAT_IXSM;

            } else {

                /* Guest zeroing out status is not a hardware requirement.

                   Clear EOP in case guest didn't do it. */

                desc.status &= ~E1000_RXD_STAT_EOP;

            }

        } else { // as per intel docs; skip descriptors with null buf addr

            DBGOUT(RX, "Null RX descriptor!!\n");

        }

        cpu_physical_memory_write(base, (void *)&desc, sizeof(desc));



        if (++s->mac_reg[RDH] * sizeof(desc) >= s->mac_reg[RDLEN])

            s->mac_reg[RDH] = 0;

        s->check_rxov = 1;

        /* see comment in start_xmit; same here */

        if (s->mac_reg[RDH] == rdh_start) {

            DBGOUT(RXERR, "RDH wraparound @%x, RDT %x, RDLEN %x\n",

                   rdh_start, s->mac_reg[RDT], s->mac_reg[RDLEN]);

            set_ics(s, 0, E1000_ICS_RXO);

            return -1;

        }

    } while (desc_offset < total_size);



    s->mac_reg[GPRC]++;

    s->mac_reg[TPR]++;

    /* TOR - Total Octets Received:

     * This register includes bytes received in a packet from the <Destination

     * Address> field through the <CRC> field, inclusively.

     */

    n = s->mac_reg[TORL] + size + /* Always include FCS length. */ 4;

    if (n < s->mac_reg[TORL])

        s->mac_reg[TORH]++;

    s->mac_reg[TORL] = n;



    n = E1000_ICS_RXT0;

    if ((rdt = s->mac_reg[RDT]) < s->mac_reg[RDH])

        rdt += s->mac_reg[RDLEN] / sizeof(desc);

    if (((rdt - s->mac_reg[RDH]) * sizeof(desc)) <= s->mac_reg[RDLEN] >>

        s->rxbuf_min_shift)

        n |= E1000_ICS_RXDMT0;



    set_ics(s, 0, n);



    return size;

}
