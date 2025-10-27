static int write_number(void *obj, const AVOption *o, void *dst, double num, int den, int64_t intnum)

{

    if (o->max*den < num*intnum || o->min*den > num*intnum) {

        av_log(obj, AV_LOG_ERROR, "Value %f for parameter '%s' out of range\n",

               num*intnum/den, o->name);

        return AVERROR(ERANGE);

    }



    switch (o->type) {

    case AV_OPT_TYPE_FLAGS:

    case AV_OPT_TYPE_INT:   *(int       *)dst= llrint(num/den)*intnum; break;

    case AV_OPT_TYPE_INT64: *(int64_t   *)dst= llrint(num/den)*intnum; break;

    case AV_OPT_TYPE_FLOAT: *(float     *)dst= num*intnum/den;         break;

    case AV_OPT_TYPE_DOUBLE:*(double    *)dst= num*intnum/den;         break;

    case AV_OPT_TYPE_RATIONAL:

        if ((int)num == num) *(AVRational*)dst= (AVRational){num*intnum, den};

        else                 *(AVRational*)dst= av_d2q(num*intnum/den, 1<<24);

        break;

    default:

        return AVERROR(EINVAL);

    }

    return 0;

}
