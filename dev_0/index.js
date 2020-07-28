const express = require('express')
const app = express()
const port = 3000
const bodyParser = require('body-parser')
const { User } = require('./models/User')

// application/x-www-form-urlenconded
app.use(bodyParser.urlencoded({extended: true}));
// applicaion/json
app.use(bodyParser.json());

// DB Config
const db = require('./config/keys').MongoURI;

// Connect to Mongo
const mongoose = require('mongoose')
mongoose.connect(db, {
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(() => console.log('MongoDB connected...'))
.catch(error => console.log(error))

app.post('/register', (req, res) => {
    // 회원 가입시 필요한 정보 client에서 가져와 DB에 넣기
    // body parser를 이용하여 req로 전송
    const user = new User(req.body)

    user.save((error, userInfo) => {
        if(error) return res.json({success: false, error})
        return res.status(200).json({
            success: true
        })
    })
})


app.get('/', (req, res) => res.send('Develog!'))
app.listen(port, () => console.log(`Example app listening on port ${port}!`))



