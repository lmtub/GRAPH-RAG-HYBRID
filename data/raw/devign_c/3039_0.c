xilinx_axienet_data_stream_push(StreamSlave *obj, uint8_t *buf, size_t size,

                                uint32_t *hdr)

{

    XilinxAXIEnetStreamSlave *ds = XILINX_AXI_ENET_DATA_STREAM(obj);

    XilinxAXIEnet *s = ds->enet;



    /* TX enable ?  */

    if (!(s->tc & TC_TX)) {

        return size;

    }



    /* Jumbo or vlan sizes ?  */

    if (!(s->tc & TC_JUM)) {

        if (size > 1518 && size <= 1522 && !(s->tc & TC_VLAN)) {

            return size;

        }

    }



    if (hdr[0] & 1) {

        unsigned int start_off = hdr[1] >> 16;

        unsigned int write_off = hdr[1] & 0xffff;

        uint32_t tmp_csum;

        uint16_t csum;



        tmp_csum = net_checksum_add(size - start_off,

                                    (uint8_t *)buf + start_off);

        /* Accumulate the seed.  */

        tmp_csum += hdr[2] & 0xffff;



        /* Fold the 32bit partial checksum.  */

        csum = net_checksum_finish(tmp_csum);



        /* Writeback.  */

        buf[write_off] = csum >> 8;

        buf[write_off + 1] = csum & 0xff;

    }



    qemu_send_packet(qemu_get_queue(s->nic), buf, size);



    s->stats.tx_bytes += size;

    s->regs[R_IS] |= IS_TX_COMPLETE;

    enet_update_irq(s);



    return size;

}
