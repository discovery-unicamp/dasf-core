export MAMBA_EXE="/usr/bin/micromamba";
export MAMBA_ROOT_PREFIX="/root/micromamba";
__mamba_setup="$('/usr/bin/micromamba' shell hook --shell bash --prefix '/root/micromamba' 2> /dev/null)"

if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    if [ -f "/root/micromamba/etc/profile.d/micromamba.sh" ]; then
        . "/root/micromamba/etc/profile.d/micromamba.sh"
    else
        export  PATH="/root/micromamba/bin:$PATH"  # extra space after export prevents interference from conda init
    fi
fi

unset __mamba_setup
