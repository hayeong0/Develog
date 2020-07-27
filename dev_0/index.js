const express = require('express')
const app = express()
const port = 3000

const mongoose = require('mongoose')
mongoose.connect('mongodb+srv://hayeong:<password>@ha0.hnyx8.mongodb.net/<dbname>?retryWrites=true&w=majority', {
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(() => console.log('MongoDB connected...'))
.catch(error => console.log(error))

app.get('/', (req, res) => res.send('Develog!'))
app.listen(port, () => console.log(`Example app listening on port ${port}!`))



