# AI portrait suite

This repo contains a full-fledged generative image app, with functionality to train a custom model on images you upload, generate new images, and display them in a gallery.

For a detailed guide on building this app from scratch, check out [our tutorial in the Interval examples gallery](https://interval.com/examples/ai-portraits).

## Running the app

* You'll need to [sign up for free at Interval](https://interval.com/signup) to get an API key
* To to train a model and generate images in a reasonable amount of time, you'll need to run the app on a GPU. We used [RunPod](https://www.runpod.io/).

```
# Close this repo
git clone https://github.com/interval/ai-portraits
cd ai-portraits

# Install dependencies and download regularization images
./setup.sh

# Start the Interval app
INTERVAL_KEY=<your key here> python main.py
```
