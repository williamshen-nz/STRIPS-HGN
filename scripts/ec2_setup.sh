set -e

# Setup expected paths
STRIPS_HGN_DIR="$HOME/strips-hgn"
SRC_DIR="$STRIPS_HGN_DIR/src"
FD_DIR="${SRC_DIR}/fast_downward"

VIRTUAL_ENV="pytorch_latest_p36"


######################
## Setup STRIPS-HGN ##
######################
# Clone the main STRIPS-HGN repo
cd $HOME
echo "Cloning STRIPS-HGN Repo from Gitlab. Please enter Username and Password."
git clone https://gitlab.com/williamshen/strips-hgn.git
cd $STRIPS_HGN_DIR

# Activate PyTorch virtual env
echo "Activating ${VIRTUAL_ENV} conda environment"
source activate $VIRTUAL_ENV

# Uninstall enum34 which breaks pyperplan and install Python dependencies
echo "Installing STRIPS-HGN Python Dependencies"
pip uninstall -y enum34
pip install -r requirements.txt

# Clone and build Fast-Downward, instructions modified from:
# http://www.fast-downward.org/ObtainingAndRunningFastDownward
git clone https://github.com/williamshen-nz/fast_downward.git $FD_DIR
cd $FD_DIR
./build.py

# Set Python path
export PYTHONPATH=$SRC_DIR
echo ""

# Change back up directories and try run the training script
cd $SRC_DIR
echo "Running train.py script to check dependencies installed correctly"
python train.py || echo "Successfully ran train.py"
echo ""

#########################
## Setup tmux sessions ##
#########################
# Deactivate conda environment otherwise tmux activation doesn't work
conda deactivate

# Get number of physical cores
NUM_PHYSICAL_CORES=`grep '^core id' /proc/cpuinfo | sort -u | wc -l`
echo "Found ${NUM_PHYSICAL_CORES} physical cores on `hostname`"
TMUX_CMD="source activate $VIRTUAL_ENV && cd $STRIPS_HGN_DIR"

# Create tmux session for each of the physical cores
for ((cpu=1; cpu<=$NUM_PHYSICAL_CORES; cpu++))
do
  session_name="cpu${cpu}"
  # Create session detach and send command to activate conda environment
  tmux new -s $session_name -d
  tmux send-keys -t $session_name "$TMUX_CMD" ENTER
  echo "Created tmux session ${session_name}"
done
echo ""

# Show how the processors are aligned
echo "Processor alignment"
cat /proc/cpuinfo | grep -E "processor|core id"
echo ""

# Start tmux session for spot instance interrupt script
echo "Starting spot interrupt script"
spot_interrupt_script="${STRIPS_HGN_DIR}/scripts/spot_interrupt.py"
tmux new -s spot-interrupt -d
tmux send-keys -t spot-interrupt \
  "python3 ${spot_interrupt_script} -e ${STRIPS_HGN_DIR}/results" ENTER


##################
## Install htop ##
##################
# May not work if the EC2 instance is new since it may still be initialising
echo "Installing htop"
{
  sudo apt install --yes htop && echo "Successfully installed htop"
} || {
  echo "Could not install htop. Try again in a few minutes."
}
