AVRational av_get_q(void *obj, const char *name, const AVOption **o_out)

{

    int64_t intnum=1;

    double num=1;

    int den=1;



    av_get_number(obj, name, o_out, &num, &den, &intnum);

    if (num == 1.0 && (int)intnum == intnum)

        return (AVRational){intnum, den};

    else

        return av_d2q(num*intnum/den, 1<<24);

}
