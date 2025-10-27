void helper_set_alarm(CPUAlphaState *env, uint64_t expire)

{

    if (expire) {

        env->alarm_expire = expire;

        qemu_mod_timer(env->alarm_timer, expire);

    } else {

        qemu_del_timer(env->alarm_timer);

    }

}
