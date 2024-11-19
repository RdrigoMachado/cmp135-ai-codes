import argparse

from SampleTWINServerBandit import SampleTWINServerBandit

# Configure the parser
parser = argparse.ArgumentParser()

parser.add_argument(
    '--debug',
    action='store_true',
    dest='debug',
    help='Run TWINS prediction service in debug mode'
)

# Parse the arguments
args = parser.parse_args()

# Create the service
server = SampleTWINServerBandit(args.debug)
server.trainModel()
