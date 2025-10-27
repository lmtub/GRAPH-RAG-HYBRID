static int my_log2(unsigned int i)

{

    unsigned int iLog2 = 0;

    while ((i >> iLog2) > 1)

	iLog2++;

    return iLog2;

}
