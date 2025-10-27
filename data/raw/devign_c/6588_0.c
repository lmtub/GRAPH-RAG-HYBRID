static int tosa_dac_send(I2CSlave *i2c, uint8_t data)

{

    TosaDACState *s = TOSA_DAC(i2c);



    s->buf[s->len] = data;

    if (s->len ++ > 2) {

#ifdef VERBOSE

        fprintf(stderr, "%s: message too long (%i bytes)\n", __FUNCTION__, s->len);

#endif

        return 1;

    }



    if (s->len == 2) {

        fprintf(stderr, "dac: channel %d value 0x%02x\n",

                s->buf[0], s->buf[1]);

    }



    return 0;

}
