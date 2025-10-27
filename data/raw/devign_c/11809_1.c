static int adb_mouse_request(ADBDevice *d, uint8_t *obuf,

                             const uint8_t *buf, int len)

{

    MouseState *s = ADB_MOUSE(d);

    int cmd, reg, olen;



    if ((buf[0] & 0x0f) == ADB_FLUSH) {

        /* flush mouse fifo */

        s->buttons_state = s->last_buttons_state;

        s->dx = 0;

        s->dy = 0;

        s->dz = 0;

        return 0;

    }



    cmd = buf[0] & 0xc;

    reg = buf[0] & 0x3;

    olen = 0;

    switch(cmd) {

    case ADB_WRITEREG:

        ADB_DPRINTF("write reg %d val 0x%2.2x\n", reg, buf[1]);

        switch(reg) {

        case 2:

            break;

        case 3:

            switch(buf[2]) {

            case ADB_CMD_SELF_TEST:

                break;

            case ADB_CMD_CHANGE_ID:

            case ADB_CMD_CHANGE_ID_AND_ACT:

            case ADB_CMD_CHANGE_ID_AND_ENABLE:

                d->devaddr = buf[1] & 0xf;

                break;

            default:

                d->devaddr = buf[1] & 0xf;

                /* we support handlers:

                 * 0x01: Classic Apple Mouse Protocol / 100 cpi operations

                 * 0x02: Classic Apple Mouse Protocol / 200 cpi operations

                 * we don't support handlers (at least):

                 * 0x03: Mouse systems A3 trackball

                 * 0x04: Extended Apple Mouse Protocol

                 * 0x2f: Microspeed mouse

                 * 0x42: Macally

                 * 0x5f: Microspeed mouse

                 * 0x66: Microspeed mouse

                 */

                if (buf[2] == 1 || buf[2] == 2) {

                    d->handler = buf[2];

                }

                break;

            }

        }

        break;

    case ADB_READREG:

        switch(reg) {

        case 0:

            olen = adb_mouse_poll(d, obuf);

            break;

        case 1:

            break;

        case 3:

            obuf[0] = d->handler;

            obuf[1] = d->devaddr;

            olen = 2;

            break;

        }

        ADB_DPRINTF("read reg %d obuf[0] 0x%2.2x obuf[1] 0x%2.2x\n", reg,

                    obuf[0], obuf[1]);

        break;

    }

    return olen;

}
