const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');

const { User } = require('./models/User');
// DB Config
const db = require('./config/keys');

// application/x-www-form-urlenconded
app.use(bodyParser.urlencoded({extended: true}));
// applicaion/json
app.use(bodyParser.json());
app.use(cookieParser());

// Connect to Mongo
const mongoose = require('mongoose');
mongoose.connect(db.mongoURI,{
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(() => console.log('MongoDB connected...'))
.catch(error => console.log(error))


app.get('/', (req, res) => res.send('Develog!'));
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

app.post('/login', (req, res) => {
    // 요청된 이메일이 데이터베이스에 있는지 찾기
    User.findOne({ email: req.body.email }, (err, user) => {
        if(!user) {
            return res.json({
                loginSuccess: false,
                message: "User not found."
            })
        }
        // 비밀번호 일치 여부 확인
        user.comparePassword(req.body.password, (err, isMatch) => {
            if(!isMatch)
            return res.json({ 
                loginSuccess: false, 
                message: "Passwords do not match."
            })
            
            // user를 위한 token 생성
            user.generateToken((err, user) => {
                if(err) return res.status(400).send(err);

                // token 저장 (쿠키)
                res.cookie("x_auth", user.token)
                .status(200)
                .json({
                    loginSuccess: true, 
                    userID: user._id})
            })
        })
    })    
})

const PORT = process.env.PORT || 3000
app.listen(PORT, () => console.log(`Example app listening on port ${PORT}!`))