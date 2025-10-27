static int do_subchannel_work(SubchDev *sch)

{

    if (sch->do_subchannel_work) {

        return sch->do_subchannel_work(sch);

    } else {

        return -EINVAL;

    }

}
