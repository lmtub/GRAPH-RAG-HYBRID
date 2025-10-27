static int ccid_bulk_in_copy_to_guest(USBCCIDState *s, uint8_t *data, int len)

{

    int ret = 0;



    assert(len > 0);

    ccid_bulk_in_get(s);

    if (s->current_bulk_in != NULL) {

        ret = MIN(s->current_bulk_in->len - s->current_bulk_in->pos, len);

        memcpy(data, s->current_bulk_in->data + s->current_bulk_in->pos, ret);

        s->current_bulk_in->pos += ret;

        if (s->current_bulk_in->pos == s->current_bulk_in->len) {

            ccid_bulk_in_release(s);

        }

    } else {

        /* return when device has no data - usb 2.0 spec Table 8-4 */

        ret = USB_RET_NAK;

    }

    if (ret > 0) {

        DPRINTF(s, D_MORE_INFO,

                "%s: %d/%d req/act to guest (BULK_IN)\n", __func__, len, ret);

    }

    if (ret != USB_RET_NAK && ret < len) {

        DPRINTF(s, 1,

            "%s: returning short (EREMOTEIO) %d < %d\n", __func__, ret, len);

    }

    return ret;

}
