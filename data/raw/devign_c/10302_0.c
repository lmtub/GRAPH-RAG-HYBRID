static void bt_hid_interrupt_sdu(void *opaque, const uint8_t *data, int len)

{

    struct bt_hid_device_s *hid = opaque;



    if (len > BT_HID_MTU || len < 1)

        goto bad;

    if ((data[0] & 3) != BT_DATA_OUTPUT)

        goto bad;

    if ((data[0] >> 4) == BT_DATA) {

        if (hid->intr_state)

            goto bad;



        hid->data_type = BT_DATA_OUTPUT;

        hid->intrdataout.len = 0;

    } else if ((data[0] >> 4) == BT_DATC) {

        if (!hid->intr_state)

            goto bad;

    } else

        goto bad;



    memcpy(hid->intrdataout.buffer + hid->intrdataout.len, data + 1, len - 1);

    hid->intrdataout.len += len - 1;

    hid->intr_state = (len == BT_HID_MTU);

    if (!hid->intr_state) {

        memcpy(hid->dataout.buffer, hid->intrdataout.buffer,

                        hid->dataout.len = hid->intrdataout.len);

        bt_hid_out(hid);

    }



    return;

bad:

    fprintf(stderr, "%s: bad transaction on Interrupt channel.\n",

                    __func__);

}
