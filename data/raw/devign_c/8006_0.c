int bitbang_i2c_set(bitbang_i2c_interface *i2c, int line, int level)

{

    int data;



    if (level != 0 && level != 1) {

        abort();

    }



    if (line == BITBANG_I2C_SDA) {

        if (level == i2c->last_data) {

            return bitbang_i2c_nop(i2c);

        }

        i2c->last_data = level;

        if (i2c->last_clock == 0) {

            return bitbang_i2c_nop(i2c);

        }

        if (level == 0) {

            DPRINTF("START\n");

            /* START condition.  */

            i2c->state = SENDING_BIT7;

            i2c->current_addr = -1;

        } else {

            /* STOP condition.  */

            bitbang_i2c_enter_stop(i2c);

        }

        return bitbang_i2c_ret(i2c, 1);

    }



    data = i2c->last_data;

    if (i2c->last_clock == level) {

        return bitbang_i2c_nop(i2c);

    }

    i2c->last_clock = level;

    if (level == 0) {

        /* State is set/read at the start of the clock pulse.

           release the data line at the end.  */

        return bitbang_i2c_ret(i2c, 1);

    }

    switch (i2c->state) {

    case STOPPED:

    case SENT_NACK:

        return bitbang_i2c_ret(i2c, 1);



    case SENDING_BIT7 ... SENDING_BIT0:

        i2c->buffer = (i2c->buffer << 1) | data;

        /* will end up in WAITING_FOR_ACK */

        i2c->state++; 

        return bitbang_i2c_ret(i2c, 1);



    case WAITING_FOR_ACK:

        if (i2c->current_addr < 0) {

            i2c->current_addr = i2c->buffer;

            DPRINTF("Address 0x%02x\n", i2c->current_addr);

            i2c_start_transfer(i2c->bus, i2c->current_addr >> 1,

                               i2c->current_addr & 1);

        } else {

            DPRINTF("Sent 0x%02x\n", i2c->buffer);

            i2c_send(i2c->bus, i2c->buffer);

        }

        if (i2c->current_addr & 1) {

            i2c->state = RECEIVING_BIT7;

        } else {

            i2c->state = SENDING_BIT7;

        }

        return bitbang_i2c_ret(i2c, 0);



    case RECEIVING_BIT7:

        i2c->buffer = i2c_recv(i2c->bus);

        DPRINTF("RX byte 0x%02x\n", i2c->buffer);

        /* Fall through... */

    case RECEIVING_BIT6 ... RECEIVING_BIT0:

        data = i2c->buffer >> 7;

        /* will end up in SENDING_ACK */

        i2c->state++;

        i2c->buffer <<= 1;

        return bitbang_i2c_ret(i2c, data);



    case SENDING_ACK:

        i2c->state = RECEIVING_BIT7;

        if (data != 0) {

            DPRINTF("NACKED\n");

            i2c->state = SENT_NACK;

            i2c_nack(i2c->bus);

        } else {

            DPRINTF("ACKED\n");

        }

        return bitbang_i2c_ret(i2c, 1);

    }

    abort();

}
