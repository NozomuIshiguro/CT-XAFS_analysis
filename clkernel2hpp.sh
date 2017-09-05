SRCPATH=$1
DEST=$2

SRC=${SRCPATH##*/}
PATH1=${SRCPATH%/*}
cd "$PATH1"
CODE=`xxd -i $SRC`
SRC=${SRC/./_}
CODE=${CODE/unsigned char $SRC[]/static string $DEST}
CODE=${CODE%unsigned int $SRC_len*}
echo "$CODE" > $SRC.hpp