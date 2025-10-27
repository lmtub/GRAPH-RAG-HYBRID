static int get_num(ByteIOContext *pb, int *len)

{

    int n, n1;



    n = get_be16(pb);

    (*len)-=2;

//    n &= 0x7FFF;

    if (n >= 0x4000) {

        return n - 0x4000;

    } else {

        n1 = get_be16(pb);

        (*len)-=2;

        return (n << 16) | n1;

    }

}
