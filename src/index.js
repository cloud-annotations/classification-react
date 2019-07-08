import React from 'react'
import ReactDOM from 'react-dom'
import MagicDropzone from 'react-magic-dropzone'

import * as tf from '@tensorflow/tfjs'
import './styles.css'

const MODEL_URL = process.env.PUBLIC_URL + '/model_web/'
const LABELS_URL = MODEL_URL + 'labels.json'
const MODEL_JSON = MODEL_URL + 'model.json'

const TFWrapper = model => {
  const detect = input => {
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(input)
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0).toFloat()
    })

    return model.execute(batched).dataSync()
  }

  return {
    detect: detect
  }
}

class App extends React.Component {
  state = {
    model: null,
    labels: null
  }

  componentDidMount() {
    const modelPromise = tf.loadGraphModel(MODEL_JSON)
    const labelsPromise = fetch(LABELS_URL).then(data => data.json())
    Promise.all([modelPromise, labelsPromise])
      .then(values => {
        const [model, labels] = values
        this.setState({
          model: model,
          labels: labels
        })
      })
      .catch(error => {
        console.error(error)
      })
  }

  cropToCanvas = (image, canvas, ctx) => {
    const naturalWidth = image.naturalWidth
    const naturalHeight = image.naturalHeight

    canvas.width = 224 // image.width
    canvas.height = 224 // image.height

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    if (naturalWidth > naturalHeight) {
      ctx.drawImage(
        image,
        (naturalWidth - naturalHeight) / 2,
        0,
        naturalHeight,
        naturalHeight,
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      )
    } else {
      ctx.drawImage(
        image,
        0,
        (naturalHeight - naturalWidth) / 2,
        naturalWidth,
        naturalWidth,
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      )
    }
  }

  onDrop = (accepted, _, links) => {
    this.setState({ preview: accepted[0].preview || links[0] })
  }

  onImageChange = e => {
    const xc = document.createElement('CANVAS')
    const xctx = xc.getContext('2d')
    this.cropToCanvas(e.target, xc, xctx)

    const predictions = TFWrapper(this.state.model).detect(xc)
    const i = predictions.indexOf(Math.max(...predictions))

    const c = document.getElementById('canvas')
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    const font = '21px sans-serif'
    ctx.font = font
    ctx.fillStyle = '#ff0000'
    ctx.fillText(this.state.labels[i], 30, 30)
  }

  render() {
    return (
      <div className="Dropzone-page">
        {this.state.model ? (
          <MagicDropzone
            className="Dropzone"
            accept="image/jpeg, image/png, .jpg, .jpeg, .png"
            multiple={false}
            onDrop={this.onDrop}
          >
            {this.state.preview ? (
              <img
                alt="upload preview"
                onLoad={this.onImageChange}
                className="Dropzone-img"
                src={this.state.preview}
              />
            ) : (
              'Choose or drop a file.'
            )}
            <canvas id="canvas" />
          </MagicDropzone>
        ) : (
          <div className="Dropzone">Loading model...</div>
        )}
      </div>
    )
  }
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
