int inet_aton (const char * str, struct in_addr * add)

{

    const char * pch = str;

    unsigned int add1 = 0, add2 = 0, add3 = 0, add4 = 0;



    add1 = atoi(pch);

    pch = strpbrk(pch,".");

    if (pch == 0 || ++pch == 0) return 0;

    add2 = atoi(pch);

    pch = strpbrk(pch,".");

    if (pch == 0 || ++pch == 0) return 0;

    add3 = atoi(pch);

    pch = strpbrk(pch,".");

    if (pch == 0 || ++pch == 0) return 0;

    add4 = atoi(pch);



    if (!add1 || (add1|add2|add3|add4) > 255) return 0;



    add->s_addr=(add4<<24)+(add3<<16)+(add2<<8)+add1;



    return 1;

}
