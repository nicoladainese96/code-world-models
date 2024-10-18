DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../data
echo "Downloading APPS data in folder: $(pwd)"

if [ -d "APPS" ]; then
  echo "APPS folder already exists. Exiting..."
  exit 1
fi

wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz # 426MB
tar -xvf APPS.tar.gz
rm APPS.tar.gz
