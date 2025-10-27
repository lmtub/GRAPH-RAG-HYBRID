static unsigned tget_long(const uint8_t **p, int le)

{

    unsigned v = le ? AV_RL32(*p) : AV_RB32(*p);

    *p += 4;

    return v;

}
